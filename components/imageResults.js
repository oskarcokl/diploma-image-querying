import { MoonLoader } from "react-spinners";
import ResultImage from "./resultImage";
import SectionTitle from "./sectionTitle";

export default function ImageResults({ srcs, styleName, onClick }) {
  return (
    <div className={`image-results-container ${styleName}`}>
      <SectionTitle title="Results" />
      <div className="result-images-container">
        <div className="column-1">
          <ResultImage src={srcs[0]} onClick={onClick}></ResultImage>
          <ResultImage src={srcs[1]} onClick={onClick}></ResultImage>
          <ResultImage src={srcs[2]} onClick={onClick}></ResultImage>
          <ResultImage src={srcs[3]} onClick={onClick}></ResultImage>
          <ResultImage src={srcs[4]} onClick={onClick}></ResultImage>
          <ResultImage src={srcs[5]} onClick={onClick}></ResultImage>
          <ResultImage src={srcs[6]} onClick={onClick}></ResultImage>
          <ResultImage src={srcs[7]} onClick={onClick}></ResultImage>
          <ResultImage src={srcs[8]} onClick={onClick}></ResultImage>
          <ResultImage src={srcs[9]} onClick={onClick}></ResultImage>
        </div>
      </div>
    </div>
  );
}
