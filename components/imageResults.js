import ResultImage from "./resultImage";
import SectionTitle from "./sectionTitle";

export default function ImageResults({ srcs, styleName }) {
  return (
    <div className={`image-results-container ${styleName}`}>
      <SectionTitle title="Results" />
      <div className="result-images-container">
        <div className="column-1">
          <ResultImage src={srcs[0]}></ResultImage>
          <ResultImage src={srcs[1]}></ResultImage>
          <ResultImage src={srcs[2]}></ResultImage>
          <ResultImage src={srcs[3]}></ResultImage>
          <ResultImage src={srcs[4]}></ResultImage>
          <ResultImage src={srcs[5]}></ResultImage>
          <ResultImage src={srcs[6]}></ResultImage>
          <ResultImage src={srcs[7]}></ResultImage>
          <ResultImage src={srcs[8]}></ResultImage>
          <ResultImage src={srcs[9]}></ResultImage>
        </div>
      </div>
    </div>
  );
}
