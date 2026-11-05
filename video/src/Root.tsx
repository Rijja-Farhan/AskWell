import {Composition} from 'remotion';
import {Launch, DURATION} from './Launch';
import {Diagram} from './Diagram';

export const Root = () => (
  <>
    <Composition id="Launch" component={Launch} durationInFrames={DURATION} fps={30} width={1920} height={1080} />
    <Composition id="Diagram" component={Diagram} durationInFrames={1} fps={30} width={1080} height={1350} />
  </>
);
