import ResultImage from "./resultImage";
import SectionTitle from "./sectionTitle";

export default function ImageResults({ srcs }) {
  return (
    <div className="image-results-container">
      <SectionTitle title="Results" />
      <div className="result-images-container">
        <div className="column-1">
          <ResultImage></ResultImage>
          <ResultImage></ResultImage>
          <ResultImage></ResultImage>
          <ResultImage></ResultImage>
          <ResultImage></ResultImage>
        </div>
        <div className="column-2">
          <ResultImage></ResultImage>
          <ResultImage></ResultImage>
          <ResultImage></ResultImage>
          <ResultImage></ResultImage>
          <ResultImage></ResultImage>
        </div>
      </div>
    </div>
  );
}
