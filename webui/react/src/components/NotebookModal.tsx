import { Button, InputNumber, Modal } from 'antd';
import { Form, Input, Select } from 'antd';
import { ModalProps } from 'antd/es/modal/Modal';
import yaml from 'js-yaml';
import React, { useCallback, useEffect, useState } from 'react';

import useStorage from 'hooks/useStorage';
import { getResourcePools, getTaskTemplates } from 'services/api';
import { NotebookConfig, RawJson, ResourcePool, ResourceType, Template } from 'types';
import { launchNotebook, previewNotebook } from 'utils/task';

import Link from './Link';
import css from './NotebookModal.module.scss';
import RadioGroup from './RadioGroup';
import Spinner from './Spinner';

const MonacoEditor = React.lazy(() => import('react-monaco-editor'));

const { Option } = Select;
const { Item } = Form;

const STORAGE_PATH = 'notebook-launch';
const STORAGE_KEY = 'notebook-config';

interface NotebookModalProps extends ModalProps {
  onLaunch?: () => void;
  visible?: boolean;
}

interface FormProps {
  fields: NotebookConfig;
  onChange: (fields: NotebookConfig) => void;
}

interface FullConfigProps {
  config: RawJson;
  onChange: (config: RawJson) => void;
}

interface ResourceInfo {
  hasCPU: boolean;
  hasGPU: boolean;
  showResourceType: boolean;
}

/*
const resourcePools = [ {
  agentDockerImage: '',
  agentDockerNetwork: '',
  agentDockerRuntime: '',
  agentFluentImage: '',
  containerStartupScript: '',
  cpuContainerCapacity: 100,
  cpuContainerCapacityPerAgent: 100,
  cpuContainersRunning: 0,
  defaultCpuPool: true,
  defaultGpuPool: true,
  description: '',
  details: {
    aws: null,
    gcp: null,
    priorityScheduler: null,
  },
  imageId: '',
  instanceType: '',
  location: 'on-prem',
  masterCertName: '',
  masterUrl: '',
  maxAgents: 0,
  maxAgentStartingPeriod: 0,
  maxIdleAgentPeriod: 0,
  minAgents: 0,
  name: 'both',
  numAgents: 1,
  preemptible: false,
  schedulerFittingPolicy: 'FITTING_POLICY_BEST',
  schedulerType: 'SCHEDULER_TYPE_FAIR_SHARE',
  slotsAvailable: 1,
  slotsPerAgent: -1,
  slotsUsed: 0,
  startupScript: '',
  type: 'RESOURCE_POOL_TYPE_STATIC',
}, {
  agentDockerImage: '',
  agentDockerNetwork: '',
  agentDockerRuntime: '',
  agentFluentImage: '',
  containerStartupScript: '',
  cpuContainerCapacity: 100,
  cpuContainerCapacityPerAgent: 100,
  cpuContainersRunning: 0,
  defaultCpuPool: true,
  defaultGpuPool: true,
  description: '',
  details: {
    aws: null,
    gcp: null,
    priorityScheduler: null,
  },
  imageId: '',
  instanceType: '',
  location: 'on-prem',
  masterCertName: '',
  masterUrl: '',
  maxAgents: 0,
  maxAgentStartingPeriod: 0,
  maxIdleAgentPeriod: 0,
  minAgents: 0,
  name: 'cpu',
  numAgents: 1,
  preemptible: false,
  schedulerFittingPolicy: 'FITTING_POLICY_BEST',
  schedulerType: 'SCHEDULER_TYPE_FAIR_SHARE',
  slotsAvailable: -1,
  slotsPerAgent: -1,
  slotsUsed: 0,
  startupScript: '',
  type: 'RESOURCE_POOL_TYPE_STATIC',
} ];
*/

const NotebookModal: React.FC<NotebookModalProps> = (
  { visible = false, onLaunch, ...props }: NotebookModalProps,
) => {

  const [ showFullConfig, setShowFullConfig ] = useState(false);
  const { fields, setFields } = useNotebookForm();
  const [ config, setConfig ] = useState<RawJson>({});

  const fetchConfig = useCallback(async () => {
    try {
      const config = await previewNotebook(
        fields.slots,
        fields.template,
        fields.name,
        fields.pool,
      );
      setConfig({ config: yaml.dump(config) });

    } catch {}
  }, [ fields ]);

  useEffect(() => {
    if (showFullConfig) fetchConfig();
  }, [ showFullConfig, fetchConfig ]);

  const handleSecondary = useCallback(() => {
    setShowFullConfig(show => !show);
  }, [ ]);

  const handleCreateEnvironment = useCallback(
    () => {
      if (showFullConfig) {
        launchNotebook(config);
      } else {
        launchNotebook(
          undefined,
          fields.slots,
          fields.template,
          fields.name,
          fields.pool,
        );
      }
      if (onLaunch) onLaunch();
    },
    [ showFullConfig, onLaunch, fields, config ],
  );

  return <Modal
    footer={<>
      <Button onClick={handleSecondary}>{showFullConfig ? 'Edit Form' : 'Edit Full Config'}</Button>
      <Button
        type="primary"
        onClick={handleCreateEnvironment}>Launch</Button>
    </>}
    title="Launch JupyterLab"
    visible={visible}
    width={540}
    {...props}>
    {showFullConfig ?
      <NotebookFullConfig config={config} onChange={(newConfig) => setConfig(newConfig)} /> :
      <NotebookForm fields={fields} onChange={(newFields) => setFields(newFields)} />
    }
  </Modal>;
};

const NotebookFullConfig:React.FC<FullConfigProps> = (
  { config, onChange }:FullConfigProps,
) => {
  const [ fields, setFields ] = useState([ { name: 'config', value: '' } ]);

  useEffect(() => {
    setFields([ { name: 'config', value: yaml.dump(config) } ]);
  }, [ config ]);

  const extractConfig = useCallback((fieldData) => {
    onChange(JSON.parse(fieldData[0].value));
  }, [ onChange ]);

  return <Form
    fields={fields}
    onFieldsChange={(_, allFields) => {
      extractConfig(allFields);
    }}>
    <div className={css.note}>
      <Link external path="/docs/reference/command-notebook-config.html">
    Read about notebook settings
      </Link>
    </div>
    <React.Suspense fallback={<div className={css.loading}><Spinner /></div>}>
      <Item
        name="config"
        noStyle
        rules={[ { message: 'Invalid YAML', required: true }, () => ({
          validator(_, value) {
            try {
              yaml.load(value);
              return Promise.resolve();
            } catch(err) {
              return Promise.reject(new Error('Invalid YAML'));
            }
          },
        }) ]}>
        <MonacoEditor
          height={430}
          language="yaml"
          options={{
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            selectOnLineNumbers: true,
          }} />
      </Item>
    </React.Suspense>
  </Form>;
};

const NotebookForm:React.FC<FormProps> = (
  { onChange, fields }: FormProps,
) => {
  const [ templates, setTemplates ] = useState<Template[]>([]);
  const [ resourcePools, setResourcePools ] = useState<ResourcePool[]>([]);
  const [ resourceInfo, setResourceInfo ] = useState<ResourceInfo>(
    { hasCPU: false, hasGPU: true, showResourceType: true },
  );

  const calculateResourceInfo = useCallback((selectedPoolName: string | undefined) => {
    const selectedPool = resourcePools.find(pool => pool.name === selectedPoolName);
    if (!selectedPool) {
      return { hasCPU: false, hasGPU: false, showResourceType: true };
    }
    const hasCPUCapacity = selectedPool.cpuContainerCapacityPerAgent > 0;
    const hasGPUCapacity = selectedPool.slotsAvailable > 0
      || (!!selectedPool.slotsPerAgent && selectedPool.slotsPerAgent > 0);
    return {
      hasCPU: hasCPUCapacity,
      hasGPU: hasGPUCapacity,
      showResourceType: hasCPUCapacity && hasGPUCapacity,
    };
  }, [ resourcePools ]);

  useEffect(() => {
    setResourceInfo(calculateResourceInfo(
      fields.pool,
    ));
  }, [ fields, calculateResourceInfo ]);

  const fetchTemplates = useCallback(async () => {
    try {
      setTemplates(await getTaskTemplates({}));
    } catch {}
  }, []);

  useEffect(() => {
    fetchTemplates();
  }, [ fetchTemplates ]);

  const fetchResourcePools = useCallback(async () => {
    try {
      setResourcePools(await getResourcePools({}));
    } catch {}
  }, []);

  useEffect(() => {
    fetchResourcePools();
  }, [ fetchResourcePools ]);

  return (<>
    <Select allowClear placeholder="No template (optional)">
      {templates.map(temp =>
        <Option key={temp.name} value={temp.name}>{temp.name}</Option>)}
    </Select>
    <Input placeholder="Name" />
    <Select
      allowClear
      placeholder="Pick the best option">
      {resourcePools.map(pool =>
        <Option key={pool.name} value={pool.name}>{pool.name}</Option>)}
    </Select>
    {resourceInfo.showResourceType &&
      <RadioGroup
        options={[ { id: ResourceType.CPU, label: ResourceType.CPU },
          { id: ResourceType.GPU, label: ResourceType.GPU } ]} />}
    {fields.type === ResourceType.GPU &&
      <InputNumber defaultValue={1} min={1} />
    }
  </>);
};

function useNotebookForm() {
  const storage = useStorage(STORAGE_PATH);
  const [ fields, setFields ] = useState<NotebookConfig>(
    storage.getWithDefault(STORAGE_KEY, { slots: 1 }),
  );

  const storeConfig = useCallback((values: NotebookConfig) => {
    delete values.name;
    storage.set(STORAGE_KEY, values);
  }, [ storage ]);

  useEffect(() => {
    storeConfig(fields);
  }, [ fields, storeConfig ]);

  return { fields, setFields };
}

export default NotebookModal;
