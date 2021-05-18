import ResultImage from "./resultImage";

export default function ImageResults({ srcs }) {
  return (
    <div className="image-results-container">
      <h2>Results</h2>
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
