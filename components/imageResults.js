import ResultImage from "./resultImage";
import SectionTitle from "./sectionTitle";

export default function ImageResults({ srcs, styleName }) {
  return (
    <div className={`image-results-container ${styleName}`}>
      <SectionTitle title="Results" />
      <div className="result-images-container">
        <div className="column-1">
          <ResultImage src={srcs[0]}></ResultImage>
          <ResultImage src={srcs[0]}></ResultImage>
          <ResultImage src={srcs[0]}></ResultImage>
          <ResultImage src={srcs[0]}></ResultImage>
          <ResultImage src={srcs[0]}></ResultImage>
        </div>
        <div className="column-2">
          <ResultImage src={srcs[0]}></ResultImage>
          <ResultImage src={srcs[0]}></ResultImage>
          <ResultImage src={srcs[0]}></ResultImage>
          <ResultImage src={srcs[0]}></ResultImage>
          <ResultImage src={srcs[0]}></ResultImage>
        </div>
      </div>
    </div>
  );
}
