import { LeftOutlined } from '@ant-design/icons';
import { Breadcrumb, Button, Dropdown, Menu, Modal, Space } from 'antd';
import React, { useCallback, useMemo, useState } from 'react';

import Avatar from 'components/Avatar';
import CopyButton from 'components/CopyButton';
import DownloadModelModal from 'components/DownloadModelModal';
import Icon from 'components/Icon';
import InfoBox, { InfoRow } from 'components/InfoBox';
import InlineEditor from 'components/InlineEditor';
import Link from 'components/Link';
import { relativeTimeRenderer, userRenderer } from 'components/Table';
import TagList from 'components/TagList';
import { useStore } from 'contexts/Store';
import { paths } from 'routes/utils';
import { ModelVersion } from 'types';
import { formatDatetime } from 'utils/date';
import { copyToClipboard } from 'utils/dom';

import css from './ModelVersionHeader.module.scss';

interface Props {
  modelVersion: ModelVersion;
  onDeregisterVersion: () => void;
  onSaveDescription: (editedNotes: string) => Promise<void>;
  onSaveName: (editedName: string) => Promise<void>;
  onUpdateTags: (newTags: string[]) => Promise<void>;
}

const ModelVersionHeader: React.FC<Props> = (
  {
    modelVersion, onDeregisterVersion,
    onSaveDescription, onUpdateTags, onSaveName,
  }: Props,
) => {
  const { auth: { user } } = useStore();
  const [ showUseInNotebook, setShowUseInNotebook ] = useState(false);
  const [ showDownloadModel, setShowDownloadModel ] = useState(false);

  const infoRows: InfoRow[] = useMemo(() => {
    return [ {
      content:
      (<Space>
        {modelVersion.username ?
          <Avatar name={modelVersion.username} /> :
          <Avatar name={modelVersion.model.username} />}
        {modelVersion.username ? modelVersion.username : modelVersion.model.username}
        on {formatDatetime(modelVersion.creationTime, 'MMM D, YYYY', false)}
      </Space>),
      label: 'Created by',
    },
    {
      content: relativeTimeRenderer(
        new Date(modelVersion.lastUpdatedTime ?? modelVersion.creationTime),
      ),
      label: 'Updated',
    },
    {
      content: <InlineEditor
        placeholder="Add description..."
        value={modelVersion.comment ?? ''}
        onSave={onSaveDescription} />,
      label: 'Description',
    },
    {
      content: <TagList
        ghost={false}
        tags={modelVersion.labels ?? []}
        onChange={onUpdateTags}
      />,
      label: 'Tags',
    } ] as InfoRow[];
  }, [ modelVersion, onSaveDescription, onUpdateTags ]);

  const referenceText = useMemo(() => {
    return (
      `from determined.experimental import Determined
model = Determined().get_model(${modelVersion?.model?.id})
ckpt = model.get_version(${modelVersion?.id}).checkpoint
ckpt_path = ckpt.download()

# WARNING: From here on out, this might not be possible to automate. Requires research.
from model import build_model
model = build_model()
model.load_state_dict(ckpt['models_state_dict'][0])

# If you get this far, you should be able to run \`model.eval()\``);
  }, [ modelVersion ]);

  const handleCopy = useCallback(async () => {
    await copyToClipboard(referenceText);
  }, [ referenceText ]);

  const isDeletable = user?.isAdmin
        || user?.username === modelVersion.model.username
        || user?.username === modelVersion.username;

  const showConfirmDelete = useCallback((version: ModelVersion) => {
    Modal.confirm({
      closable: true,
      content: `Are you sure you want to delete this version "Version ${version.version}" 
      from this model?`,
      icon: null,
      maskClosable: true,
      okText: 'Delete Version',
      okType: 'danger',
      onOk: () => onDeregisterVersion(),
      title: 'Confirm Delete',
    });
  }, [ onDeregisterVersion ]);

  return (
    <header className={css.base}>
      <div className={css.breadcrumbs}>
        <Breadcrumb separator="">
          <Breadcrumb.Item>
            <Link path={paths.modelDetails(modelVersion.model.id)}>
              <LeftOutlined style={{ marginRight: 10 }} />
            </Link>
          </Breadcrumb.Item>
          <Breadcrumb.Item>
            <Link path={paths.modelList()}>
              Model Registry
            </Link>
          </Breadcrumb.Item>
          <Breadcrumb.Separator />
          <Breadcrumb.Item>
            <Link path={paths.modelDetails(modelVersion.model.id)}>
              {modelVersion.model.name}
            </Link>
          </Breadcrumb.Item>
          <Breadcrumb.Separator />
          <Breadcrumb.Item>Version {modelVersion.version}</Breadcrumb.Item>
        </Breadcrumb>
      </div>
      <div className={css.headerContent}>
        <div className={css.mainRow}>
          <div className={css.title}>
            <div className={css.versionBox}>
              V{modelVersion.version}
            </div>
            <h1 className={css.versionName}>
              <InlineEditor
                placeholder="Add name..."
                value = {modelVersion.name ? modelVersion.name : `Version ${modelVersion.version}`}
                onSave={onSaveName}
              />
            </h1>
          </div>
          <div className={css.buttons}>
            <Button onClick={() => setShowDownloadModel(true)}>Download Model</Button>
            <DownloadModelModal
              modelVersion={modelVersion}
              visible={showDownloadModel}
              onClose={() => setShowDownloadModel(false)} />
            <Button onClick={() => setShowUseInNotebook(true)}>Use in Notebook</Button>
            <Modal
              className={css.useNotebookModal}
              footer={null}
              title="Use in Notebook"
              visible={showUseInNotebook}
              onCancel={() => setShowUseInNotebook(false)}>
              <div className={css.topLine}>
                <p>Reference this model in a notebook</p>
                <CopyButton onCopy={handleCopy} />
              </div>
              <pre className={css.codeSample}><code>{referenceText}</code></pre>
              <p>Copy/paste code into a notebook cell</p>
            </Modal>
            <Dropdown
              overlay={(
                <Menu>
                  <Menu.Item
                    danger
                    disabled={!isDeletable}
                    key="deregister-version"
                    onClick={() => showConfirmDelete(modelVersion)}>
                  Deregister Version
                  </Menu.Item>
                </Menu>
              )}
              trigger={[ 'click' ]}>
              <Button type="text">
                <Icon name="overflow-horizontal" size="tiny" />
              </Button>
            </Dropdown>
          </div>
        </div>
        <InfoBox rows={infoRows} separator={false} />
      </div>
    </header>
  );
};

export default ModelVersionHeader;
