import {Composition} from 'remotion';
import {Launch, DURATION} from './Launch';
import {Launch2, DURATION2} from './Launch2';
import {Flow, DURATION_FLOW} from './Flow';
import {Diagram} from './Diagram';

export const Root = () => (
  <>
    <Composition id="Launch" component={Launch} durationInFrames={DURATION} fps={30} width={1920} height={1080} />
    <Composition id="Launch2" component={Launch2} durationInFrames={DURATION2} fps={30} width={1920} height={1080} />
    <Composition id="Flow" component={Flow} durationInFrames={DURATION_FLOW} fps={30} width={1920} height={1080} />
    <Composition id="Diagram" component={Diagram} durationInFrames={1} fps={30} width={1080} height={1350} />
  </>
);
