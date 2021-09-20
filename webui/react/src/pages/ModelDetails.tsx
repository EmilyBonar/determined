import { Button, Tooltip } from 'antd';
import { ColumnsType } from 'antd/lib/table';
import React, { useCallback, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';

import Icon from 'components/Icon';
import InfoBox from 'components/InfoBox';
import Message, { MessageType } from 'components/Message';
import Page from 'components/Page';
import ResponsiveTable from 'components/ResponsiveTable';
import Spinner from 'components/Spinner';
import { modelVersionNameRenderer, relativeTimeRenderer } from 'components/Table';
import usePolling from 'hooks/usePolling';
import { getModelDetails } from 'services/api';
import { V1GetModelVersionsRequestSortBy } from 'services/api-ts-sdk';
import { isAborted, isNotFound } from 'services/utils';
import { ModelVersion, ModelVersions } from 'types';
import { isEqual } from 'utils/data';

import CollapsableCard from './ModelDetails/CollapsableCard';
import ModelHeader from './ModelDetails/ModelHeader';

interface Params {
  modelId: string;
}

const ModelDetails: React.FC = () => {
  const [ model, setModel ] = useState<ModelVersions>();
  const { modelId } = useParams<Params>();
  const [ pageError, setPageError ] = useState<Error>();

  const id = parseInt(modelId);

  const fetchModel = useCallback(async () => {
    try {
      const modelData = await getModelDetails({ modelName: 'mnist', sortBy: 'SORT_BY_VERSION' });
      if (!isEqual(modelData, model)) setModel(modelData);
    } catch (e) {
      if (!pageError && !isAborted(e)) setPageError(e as Error);
    }
  }, [ model, pageError ]);

  usePolling(fetchModel);

  const columns = useMemo(() => {
    const tableColumns: ColumnsType<ModelVersion> = [
      {
        dataIndex: 'version',
        key: V1GetModelVersionsRequestSortBy.VERSION,
        sorter: true,
        title: 'Version',
      },
      {
        dataIndex: 'name',
        render: modelVersionNameRenderer,
        title: 'Name',
      },
      {
        dataIndex: 'description',
        title: 'Description',
      },
      {
        dataIndex: 'lastUpdatedTime',
        render: relativeTimeRenderer,
        sorter: true,
        title: 'Last updated',
      },
      { dataIndex: 'tags', title: 'Tags' },
    ];

    return tableColumns;
  }, []);

  const metadata = useMemo(() => {
    return Object.entries(model?.model.metadata || {}).map((pair) => {
      return ({ content: pair[1], label: pair[0] });
    });
  }, [ model?.model. metadata ]);

  const referenceText = useMemo(() => {
    return (
      `from determined.experimental import Determined
model = Determined.getModel("${model?.model.name}")
ckpt = model.get_version("1234")
ckpt_path = ckpt.download()
ckpt = torch.load(os.path.join(ckpt_path, 'state_dict.pth'))

# WARNING: From here on out, this might not be possible to automate. Requires research.
from model import build_model
model = build_model()
model.load_state_dict(ckpt['models_state_dict'][0])

# If you get this far, you should be able to run \`model.eval()\``);
  }, [ model?.model.name ]);

  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(referenceText);
  }, [ referenceText ]);

  if (isNaN(id)) {
    return <Message title={`Invalid Model ID ${modelId}`} />;
  } else if (pageError) {
    const message = isNotFound(pageError) ?
      `Unable to find model ${modelId}` :
      `Unable to fetch model ${modelId}`;
    return <Message title={message} type={MessageType.Warning} />;
  } else if (!model) {
    return <Spinner tip={`Loading model ${modelId} details...`} />;
  }

  return (
    <Page
      docTitle="Model Details"
      headerComponent={<ModelHeader model={model.model} />}
      id="modelDetails">
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        marginLeft: 20,
        marginRight: 20,
      }}>{
          model.modelVersions.length === 0 ?
            <div style={{
              alignItems: 'center',
              display: 'flex',
              flexDirection: 'column',
              margin: 'var(--theme-sizes-layout-huge)',
            }}>
              <p>No Model Versions</p>
              <p style={{
                color: 'var(--theme-colors-monochrome-9)',
                fontSize: 'var(--theme-sizes-font-medium',
                maxWidth: '370px',
                textAlign: 'center',
              }}>
                Register a checkpoint from an experiment to add it to this model
              </p>
            </div> :
            <ResponsiveTable
              columns={columns}
              dataSource={model.modelVersions}
              showSorterTooltip={false} />
        }
        {metadata.length > 0 &&
        <CollapsableCard title={'Metadata'}>
          <InfoBox rows={metadata} />
          <Button type="link">add row</Button>
        </CollapsableCard>
        }
        <CollapsableCard
          extra={(
            <Tooltip title="Copied!" trigger="click">
              <Button type="link" onClick={handleCopy}>Copy to clipboard</Button>
            </Tooltip>
          )}
          title={<>How to reference this model <Icon name="info" /></>}>
          <pre>{referenceText}</pre>
        </CollapsableCard>
      </div>
    </Page>
  );
};

export default ModelDetails;
