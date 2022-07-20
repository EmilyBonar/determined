import { InfoCircleOutlined } from '@ant-design/icons';
import { Alert, Button, Checkbox, Form, Input, InputNumber, ModalFuncProps,
  Radio, RadioChangeEvent, Select, Space, Tooltip, Typography } from 'antd';
import { CheckboxChangeEvent } from 'antd/lib/checkbox';
import { SelectValue } from 'antd/lib/select';
import yaml from 'js-yaml';
import React, { useCallback, useEffect, useMemo, useState } from 'react';

import Link from 'components/Link';
import SelectFilter from 'components/SelectFilter';
import { useStore } from 'contexts/Store';
import { maxPoolSlotCapacity } from 'pages/Clusters/ClustersOverview';
import { paths } from 'routes/utils';
import { createExperiment } from 'services/api';
import Icon from 'shared/components/Icon';
import { Primitive } from 'shared/types';
import { clone, flattenObject, isBoolean, unflattenObject } from 'shared/utils/data';
import { DetError, isDetError } from 'shared/utils/error';
import { roundToPrecision } from 'shared/utils/number';
import { routeToReactUrl } from 'shared/utils/routes';
import { validateLength } from 'shared/utils/string';
import { ExperimentBase, ExperimentSearcherName, Hyperparameter,
  HyperparameterType, ResourcePool, TrialDetails, TrialHyperparameters, TrialItem } from 'types';

import useModal, { ModalHooks as Hooks, ModalCloseReason } from './useModal';
import css from './useModalHyperparameterSearch.module.scss';

interface Props {
  experiment: ExperimentBase;
  trial?: TrialDetails | TrialItem;
}

export interface ShowModalProps {
  initialModalProps?: ModalFuncProps;
  trial?: TrialDetails | TrialItem;
}

interface ModalHooks extends Omit<Hooks, 'modalOpen'> {
  modalOpen: (props?: ShowModalProps) => void;
}

interface SearchMethod {
  displayName: string;
  icon: React.ReactNode;
  name: `${ExperimentSearcherName}`;
}

const SearchMethods: Record<string, SearchMethod> = {
  ASHA: {
    displayName: 'Adaptive',
    icon: <Icon name="searcher-adaptive" />,
    name: 'adaptive_asha',
  },
  Grid: {
    displayName: 'Grid',
    icon: <Icon name="searcher-grid" />,
    name: 'grid',
  },
  Random: {
    displayName: 'Random',
    icon: <Icon name="searcher-random" />,
    name: 'random',
  },
} as const;

interface HyperparameterRowValues {
  count?: number;
  max?: number,
  min?:number
  type: HyperparameterType,
  value?: number | string,
}

const useModalHyperparameterSearch = ({ experiment, trial: trialIn }: Props): ModalHooks => {
  const { modalClose, modalOpen: openOrUpdate, modalRef, contextHolder } = useModal();
  const [ trial, setTrial ] = useState(trialIn);
  const [ modalError, setModalError ] = useState<string>();
  const [ searcher, setSearcher ] = useState(SearchMethods.ASHA);
  const { resourcePools } = useStore();
  const [ resourcePool, setResourcePool ] = useState<ResourcePool>();
  const [ form ] = Form.useForm();
  const [ currentPage, setCurrentPage ] = useState(0);
  const [ canceler ] = useState(new AbortController());
  const [ validationError, setValidationError ] = useState(false);
  const formValues = Form.useWatch([], form);

  const trialHyperparameters = useMemo(() => {
    if (!trial) return;
    const continueFn = (value: unknown) => value === 'object';
    return flattenObject<TrialHyperparameters>(
      trial.hyperparameters,
      { continueFn },
    ) as unknown as Record<string, Primitive>;
  }, [ trial ]);

  const hyperparameters = useMemo(() => {
    return Object.entries(experiment.hyperparameters).map(hp => {
      const hpObject = ({ hyperparameter: hp[1], name: hp[0] });
      if (trialHyperparameters?.[hp[0]]) {
        hpObject.hyperparameter.val = trialHyperparameters[hp[0]];
      }
      return hpObject;
    });
  }, [ experiment.hyperparameters, trialHyperparameters ]);

  const submitExperiment = useCallback(async () => {
    const fields: Record<string, Primitive | HyperparameterRowValues> = form.getFieldsValue(true);

    const baseConfig = clone(experiment.configRaw, true);
    baseConfig.name = (fields.name as string).trim();
    baseConfig.searcher.name = fields.searcher;
    baseConfig.searcher.max_trials = fields.searcher === SearchMethods.Grid.name ?
      undefined : fields.max_trials;
    baseConfig.searcher.max_length[fields.length_units as string] = fields.max_length;
    baseConfig.searcher.metric = fields.metric;
    baseConfig.searcher.smaller_is_better = fields.smaller_is_better;
    baseConfig.resources.resource_pool = fields.pool;
    baseConfig.resources.max_slots = fields.slots;
    baseConfig.searcher.bracket_rungs = undefined;

    if (fields.searcher === SearchMethods.ASHA.name) {
      baseConfig.searcher.stop_once = baseConfig.stop_once ?? false;
      baseConfig.searcher.max_rungs = baseConfig.max_rungs ?? 5;
      baseConfig.searcher.divisor = baseConfig.divisor ?? 4;
      baseConfig.searcher.mode = baseConfig.mode ?? 'standard';
    } else if (fields.searcher === SearchMethods.AsyncHalving.name) {
      baseConfig.searcher.stop_once = baseConfig.stop_once ?? false;
      baseConfig.searcher.max_rungs = undefined;
      baseConfig.searcher.divisor = baseConfig.divisor ?? 4;
      baseConfig.searcher.mode = undefined;
    } else {
      baseConfig.searcher.stop_once = undefined;
      baseConfig.searcher.max_rungs = undefined;
      baseConfig.searcher.divisor = undefined;
      baseConfig.searcher.mode = undefined;
    }

    Object.entries(fields)
      .filter(field => typeof field[1] === 'object')
      .forEach(hp => {
        const hpName = hp[0];
        const hpInfo = hp[1] as HyperparameterRowValues;
        if (hpInfo.type === HyperparameterType.Categorical) return;
        else if (hpInfo.type === HyperparameterType.Constant) {
          if (!hpInfo.value) return;
          let parsedVal;
          try {
            if (typeof hpInfo.value === 'string'){
              parsedVal = JSON.parse(hpInfo.value);
            } else {
              parsedVal = hpInfo.value;
            }
          } catch (e) {
            parsedVal = hpInfo.value;
          }
          baseConfig.hyperparameters[hpName] = {
            type: hpInfo.type,
            val: parsedVal,
          };
        } else {
          baseConfig.hyperparameters[hpName] = {
            count: fields.searcher === SearchMethods.Grid.name ? hpInfo.count : undefined,
            maxval: hpInfo.type === HyperparameterType.Int ?
              roundToPrecision(hpInfo.max ?? 0, 0) :
              hpInfo.max,
            minval: hpInfo.type === HyperparameterType.Int ?
              roundToPrecision(hpInfo.min ?? 0, 0) :
              hpInfo.min,
            type: hpInfo.type,
          };
        }
        if (hpInfo.type === HyperparameterType.Log) baseConfig.hyperparameters[hpName].base = 10.0;
      });

    baseConfig.hyperparameters = unflattenObject(baseConfig.hyperparameters);

    const newConfig = yaml.dump(baseConfig);

    try {
      const { id: newExperimentId } = await createExperiment({
        activate: true,
        experimentConfig: newConfig,
        parentId: experiment.id,
        projectId: experiment.projectId,
      }, { signal: canceler.signal });

      // Route to reload path to forcibly remount experiment page.
      const newPath = paths.experimentDetails(newExperimentId);
      routeToReactUrl(paths.reload(newPath));
    } catch (e) {
      let errorMessage = 'Unable to create experiment.';
      if (isDetError(e)) {
        errorMessage = e.publicMessage || e.message;
      }

      setModalError(errorMessage);

      // We throw an error to prevent the modal from closing.
      throw new DetError(errorMessage, { publicMessage: errorMessage, silent: true });
    }
  }, [ canceler.signal, experiment.configRaw, experiment.id, experiment.projectId, form ]);

  const handleOk = useCallback(() => {
    if (currentPage === 0) {
      setCurrentPage(1);
    } else {
      submitExperiment();
    }
  }, [ currentPage, submitExperiment ]);

  const handleBack = useCallback(() => {
    setCurrentPage(prev => prev - 1);
  }, []);

  const handleCancel = useCallback(() => {
    modalClose(ModalCloseReason.Cancel);
  }, [ modalClose ]);

  const handleSelectPool = useCallback((value: SelectValue) => {
    setResourcePool(resourcePools.find(pool => pool.name === value));
  }, [ resourcePools ]);

  const maxSlots = useMemo(
    () => resourcePool ? maxPoolSlotCapacity(resourcePool) : 0,
    [ resourcePool ],
  );

  const [ maxLengthUnit, maxLength ] = useMemo(() => {
    return Object.entries(
      experiment.configRaw.searcher.max_length,
    )[0] as ['batches' | 'records' | 'epochs', number];
  }, [ experiment.configRaw.searcher.max_length ]);

  useEffect(() => {
    if (resourcePool || resourcePools.length === 0) return;
    setResourcePool(resourcePools[0]);
  }, [ resourcePool, resourcePools ]);

  const validateForm = useCallback(() => {
    if (!formValues) return;
    if (currentPage === 1) {
      const hyperparameters = formValues as Record<string, HyperparameterRowValues>;
      setValidationError(!Object.values(hyperparameters).every(hp => {
        switch (hp.type) {
          case HyperparameterType.Constant:
          case HyperparameterType.Categorical:
            return hp.value != null;
          default:
            return hp.min != null && hp.max != null && hp.max >= hp.min &&
              (searcher !== SearchMethods.Grid || (hp.count != null && hp.count > 0));
        }
      }));
    } else if (currentPage === 0) {
      const {
        searcher, name, pool, slots_per_trial, max_trials, max_length,
        length_units, mode, stop_once, max_concurrent_trials,
      } = formValues;
      setValidationError(!(validateLength(name ?? '') && slots_per_trial != null &&
        slots_per_trial >= 0 && slots_per_trial <= maxSlots && max_length != null &&
        max_length > 0 && max_concurrent_trials != null && max_concurrent_trials >= 0 &&
        (searcher === SearchMethods.Grid.name || (max_trials != null && max_trials > 0)) &&
        pool != null && length_units != null &&
        (searcher !== SearchMethods.ASHA.name || (mode != null && isBoolean(stop_once)))
      ));
    }
  }, [ currentPage, formValues, maxSlots, searcher ]);

  useEffect(() => {
    validateForm();
  }, [ validateForm ]);

  const handleSelectSearcher = useCallback((e: RadioChangeEvent) => {
    const value = e.target.value;
    setSearcher(
      Object.values(SearchMethods).find(searcher => searcher.name === value) ?? SearchMethods.ASHA,
    );
  }, []);

  const hyperparameterPage = useMemo((): React.ReactNode => {
    // We always render the form regardless of mode to provide a reference to it.
    return (
      <div className={css.base}>
        {modalError && <Alert className={css.error} message={modalError} type="error" />}
        <div
          className={css.hyperparameterContainer}
          style={{
            gridTemplateColumns: `180px minmax(0, 1.4fr) 
              repeat(${searcher === SearchMethods.Grid ? 4 : 3}, minmax(0, 1fr))`,
          }}>
          <h2 className={css.hyperparameterName}>Hyperparameter</h2>
          <h2>Type</h2>
          <h2>Current</h2>
          <h2>Min Value</h2>
          <h2>Max Value</h2>
          {searcher === SearchMethods.Grid && <h2>Grid Count</h2>}
          {hyperparameters.map(hp => (
            <HyperparameterRow
              key={hp.name}
              searcher={searcher}
              {...hp}
            />
          ))}
        </div>
      </div>
    );
  }, [ hyperparameters, modalError, searcher ]);

  const searcherPage = useMemo((): React.ReactNode => {
    // We always render the form regardless of mode to provide a reference to it.
    return (
      <div className={css.base}>
        {modalError && <Alert className={css.error} message={modalError} type="error" />}
        <Form.Item
          initialValue={searcher.name}
          label={(
            <div className={css.searchTitle}>
              <p>Select search method</p>
              <Link
                external
                path={paths.
                  docs('/training-hyperparameter/index.html#specifying-the-search-algorithm')}>
                Learn more
              </Link>
            </div>
          )}
          name="searcher">
          <Radio.Group
            className={css.searcherGroup}
            optionType="button"
            onChange={handleSelectSearcher}>
            {Object.values(SearchMethods).map(searcher => (
              <Radio.Button
                key={searcher.name}
                value={searcher.name}>
                <div className={css.searcherButton}>
                  {searcher.icon}
                  <p>{searcher.displayName}</p>
                </div>
              </Radio.Button>
            ))}
          </Radio.Group>
        </Form.Item>
        <Form.Item
          initialValue={experiment.name}
          label="New Experiment Name"
          name="name"
          rules={[ { required: true } ]}>
          <Input maxLength={80} />
        </Form.Item>
        <Form.Item
          initialValue={resourcePools?.[0]?.name}
          label="Resource Pool"
          name="pool"
          rules={[ { required: true } ]}>
          <SelectFilter
            showSearch={false}
            onChange={handleSelectPool}>
            {resourcePools.map(pool => (
              <Select.Option key={pool.name} value={pool.name}>
                {pool.name}
              </Select.Option>
            ))}
          </SelectFilter>
        </Form.Item>
        <p>{maxSlots} max slots</p>
        <h2 className={css.sectionTitle}>Configure Trials</h2>
        <div className={css.inputRow}>
          <Form.Item
            initialValue={maxLength}
            label="Max Length"
            name="max_length"
            rules={[ { min: 1, required: true, type: 'number' } ]}>
            <InputNumber precision={0} />
          </Form.Item>
          <Form.Item
            initialValue={maxLengthUnit}
            label="Units"
            name="length_units"
            rules={[ { required: true } ]}>
            <SelectFilter
              onChange={handleSelectPool}>
              <Select.Option value="records">
                records
              </Select.Option>
              <Select.Option value="batches">
                batches
              </Select.Option>
              {(experiment.configRaw?.records_per_epoch ?? 0) > 0 && (
                <Select.Option value="epochs">
                  epochs
                </Select.Option>
              )}
            </SelectFilter>
          </Form.Item>
          <Form.Item
            initialValue={experiment.configRaw?.resources?.slots_per_trial ??
              maxSlots === 0 ? 0 : 1}
            label="Slots per trial"
            name="slots_per_trial"
            rules={[ { max: maxSlots, min: 0, required: true, type: 'number' } ]}>
            <InputNumber precision={0} />
          </Form.Item>
        </div>
        {searcher.name === 'adaptive_asha' && (
          <Form.Item
            initialValue={experiment.configRaw.searcher?.mode ?? 'standard'}
            label={(
              <div className={css.labelWithTooltip}>
                Early Stopping Mode
                <Tooltip
                  title="How aggressively to perform early stopping of underperforming trials">
                  <InfoCircleOutlined />
                </Tooltip>
              </div>
            )}
            name="mode"
            rules={[ { required: true } ]}>
            <SelectFilter showSearch={false}>
              <Select.Option value="aggressive">
                Aggressive
              </Select.Option>
              <Select.Option value="standard">
                Standard
              </Select.Option>
              <Select.Option value="conservative">
                Conservative
              </Select.Option>
            </SelectFilter>
          </Form.Item>
        )}
        {searcher.name === 'adaptive_asha' && (
          <Form.Item
            initialValue={experiment.configRaw.searcher?.stop_once ?? true}
            name="stop_once"
            rules={[ { required: true } ]}
            valuePropName="checked">
            <Checkbox>
              Stop once - Only stop trials one time when there is enough evidence to
              terminate training (recommended for faster search)
            </Checkbox>
          </Form.Item>
        )}
        <h2 className={css.sectionTitle}>Set Training Limits</h2>
        <div className={css.inputRow}>
          <Form.Item
            hidden={searcher === SearchMethods.Grid}
            initialValue={1}
            label="Max trials"
            name="max_trials"
            rules={[ { min: 1, required: true, type: 'number' } ]}>
            <InputNumber precision={0} />
          </Form.Item>
          <Form.Item
            initialValue={0}
            label={(
              <div className={css.labelWithTooltip}>
                Max concurrent trials
                <Tooltip title="Use 0 for max possible parallelism">
                  <InfoCircleOutlined style={{ color: 'var(--theme-colors-monochrome-8)' }} />
                </Tooltip>
              </div>
            )}
            name="max_concurrent_trials"
            rules={[ { min: 0, required: true, type: 'number' } ]}>
            <InputNumber precision={0} />
          </Form.Item>
        </div>
      </div>
    );
  }, [ experiment.configRaw?.records_per_epoch,
    experiment.configRaw?.resources?.slots_per_trial,
    experiment.configRaw.searcher?.mode,
    experiment.configRaw.searcher?.stop_once,
    experiment.name,
    handleSelectPool,
    handleSelectSearcher,
    maxLength,
    maxLengthUnit,
    maxSlots,
    modalError,
    resourcePools,
    searcher ]);

  const pages = useMemo(
    () => [ searcherPage, hyperparameterPage ],
    [ hyperparameterPage, searcherPage ],
  );

  const footer = useMemo(() => {
    return (
      <div className={css.footer}>
        {currentPage > 0 && <Button onClick={handleBack}>Back</Button>}
        <div className={css.spacer} />
        <Space>
          <Button onClick={handleCancel}>Cancel</Button>
          <Button disabled={validationError} type="primary" onClick={handleOk}>
            {currentPage === 0 ? 'Select Hyperparameters' : 'Run Experiment'}
          </Button>
        </Space>
      </div>
    );
  }, [ currentPage, handleBack, handleCancel, handleOk, validationError ]);

  const modalProps: Partial<ModalFuncProps> = useMemo(() => {
    return {
      className: css.modal,
      closable: true,
      content: (
        <Form form={form} layout="vertical" requiredMark={false}>
          {pages[currentPage]}
          {footer}
        </Form>),
      icon: null,
      title: 'Hyperparameter Search',
      width: 700,
    };
  }, [ form, pages, currentPage, footer ]);

  const modalOpen = useCallback((props?: ShowModalProps) => {
    setCurrentPage(0);
    form.resetFields();
    if (props?.trial) setTrial(props?.trial);
    openOrUpdate({ ...modalProps, ...props?.initialModalProps });
  }, [ form, modalProps, openOrUpdate ]);

  /*
   * When modal props changes are detected, such as modal content
   * title, and buttons, update the modal
   */
  useEffect(() => {
    if (modalRef.current) openOrUpdate(modalProps);
  }, [ modalProps, modalRef, openOrUpdate ]);

  return { contextHolder, modalClose, modalOpen, modalRef };
};

interface RowProps {
  hyperparameter: Hyperparameter;
  name: string;
  searcher: SearchMethod;
}

const HyperparameterRow: React.FC<RowProps> = (
  { hyperparameter, name, searcher }: RowProps,
) => {
  const form = Form.useFormInstance();
  const type: HyperparameterType | undefined = Form.useWatch([ name, 'type' ]);
  const checked: boolean | undefined = Form.useWatch([ name, 'active' ]);
  const min: number | undefined = Form.useWatch([ name, 'min' ]);
  const max: number | undefined = Form.useWatch([ name, 'max' ]);
  const [ valError, setValError ] = useState<string>();
  const [ minError, setMinError ] = useState<string>();
  const [ maxError, setMaxError ] = useState<string>();
  const [ rangeError, setRangeError ] = useState<string>();
  const [ countError, setCountError ] = useState<string>();

  const handleCheck = useCallback((e: CheckboxChangeEvent) => {
    const { checked } = e.target;
    form.setFields([ {
      name: [ name, 'type' ],
      value: checked ? HyperparameterType.Double : HyperparameterType.Constant,
    } ]);
  }, [ form, name ]);

  const handleTypeChange = useCallback((value: HyperparameterType) => {
    form.setFields([ {
      name: [ name, 'active' ],
      value: value !== HyperparameterType.Constant,
    } ]);
  }, [ form, name ]);

  const validateValue = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const { value } = e.target;
    setValError(value === '' ? 'Current value is required.' : undefined);
  }, []);

  const validateMin = useCallback((value: number | string | null) => {
    if (value == null) {
      setMinError('Minimum value is required.');
    } else if (typeof value === 'string') {
      setMinError('Minimum value must be a number.');
    } else if (max != null && value > max) {
      setRangeError('Maximum value must be greater or equal to than minimum value.');
      setMinError(undefined);
    } else {
      setMinError(undefined);
      setRangeError(undefined);
    }
  }, [ max ]);

  const validateMax = useCallback((value: number | string | null) => {
    if (value == null) {
      setMaxError('Maximum value is required.');
    } else if (typeof value === 'string') {
      setMaxError('Maximum value must be a number.');
    } else if (min != null && value < min) {
      setRangeError('Maximum value must be greater or equal to than minimum value.');
    } else {
      setMaxError(undefined);
      setRangeError(undefined);
    }
  }, [ min ]);

  const validateCount = useCallback((value: number | string | null) => {
    if (value == null) {
      setCountError('Grid count is required.');
    } else if (typeof value === 'string') {
      setCountError('Grid count must be a number.');
    } else if (value < 1) {
      setCountError('Grid count must be greater than or equal to 1.');
    } else {
      setCountError(undefined);
    }
  }, []);

  return (
    <>
      <Space className={css.hyperparameterName}>
        <Form.Item
          initialValue={hyperparameter.type !== HyperparameterType.Constant}
          name={[ name, 'active' ]}
          noStyle
          valuePropName="checked">
          <Checkbox onChange={handleCheck} />
        </Form.Item>
        <Typography.Title ellipsis={{ rows: 1, tooltip: true }} level={3}>{name}</Typography.Title>
      </Space>
      <Form.Item
        initialValue={hyperparameter.type}
        name={[ name, 'type' ]}
        noStyle>
        <Select onChange={handleTypeChange}>
          {(Object.keys(HyperparameterType) as Array<keyof typeof HyperparameterType>)
            .map((type) => (
              <Select.Option
                disabled={HyperparameterType[type] === HyperparameterType.Categorical}
                key={HyperparameterType[type]}
                value={HyperparameterType[type]}>
                {type}
                {type === 'Log' ? ' (base 10)' : ''}
              </Select.Option>
            ))}
        </Select>
      </Form.Item>
      <Form.Item
        initialValue={hyperparameter.val}
        name={[ name, 'value' ]}
        rules={[ { required: !checked } ]}
        validateStatus={valError ? 'error' : 'success'}>
        <Input
          disabled={checked}
          placeholder={checked ? 'n/a' : ''}
          onChange={validateValue}
        />
      </Form.Item>
      {type === HyperparameterType.Categorical ? (
        <>
          <Form.Item
            initialValue={hyperparameter.minval}
            name={[ name, 'min' ]}
            noStyle>
            <Input disabled />
          </Form.Item>
          <Form.Item
            initialValue={hyperparameter.maxval}
            name={[ name, 'max' ]}
            noStyle>
            <Input disabled />
          </Form.Item>
          <Form.Item
            hidden={searcher !== SearchMethods.Grid}
            name={[ name, 'count' ]}>
            <InputNumber disabled />
          </Form.Item>
        </>
      ) : (
        <>
          <Form.Item
            initialValue={hyperparameter.minval}
            name={[ name, 'min' ]}
            rules={[ {
              max: max,
              required: checked,
              type: 'number',
            } ]}
            validateStatus={((minError || rangeError) && checked) ? 'error' : undefined}>
            <InputNumber
              disabled={!checked}
              placeholder={!checked ? 'n/a' : ''}
              precision={type === HyperparameterType.Int ? 0 : undefined}
              onChange={validateMin}
            />
          </Form.Item>
          <Form.Item
            initialValue={hyperparameter.maxval}
            name={[ name, 'max' ]}
            rules={[ {
              min: min,
              required: checked,
              type: 'number',
            } ]}
            validateStatus={((maxError || rangeError) && checked) ? 'error' : undefined}>
            <InputNumber
              disabled={!checked}
              placeholder={!checked ? 'n/a' : ''}
              precision={type === HyperparameterType.Int ? 0 : undefined}
              onChange={validateMax}
            />
          </Form.Item>
          <Form.Item
            hidden={searcher !== SearchMethods.Grid}
            name={[ name, 'count' ]}
            rules={[ {
              min: 0,
              required: checked && searcher === SearchMethods.Grid,
              type: 'number',
            } ]}
            validateStatus={(typeof countError === 'string' && searcher === SearchMethods.Grid
                && checked) ? 'error' : undefined}>
            <InputNumber
              disabled={!checked}
              placeholder={!checked ? 'n/a' : ''}
              precision={0}
              onChange={validateCount}
            />
          </Form.Item>
        </>
      )}
      {type === HyperparameterType.Categorical &&
        <p className={css.warning}>Categorical hyperparameters are not currently supported.</p>}
      {(!checked && valError) &&
        <p className={css.error}>{valError}</p>}
      {(checked && minError) &&
        <p className={css.error}>{minError}</p>}
      {(checked && maxError) &&
        <p className={css.error}>{maxError}</p>}
      {(checked && rangeError) &&
        <p className={css.error}>{rangeError}</p>}
      {(checked && countError && searcher === SearchMethods.Grid) &&
        <p className={css.error}>{countError}</p>}
    </>
  );
};

export default useModalHyperparameterSearch;
