import ResultImage from "./resultImage";
import SectionTitle from "./sectionTitle";
import Button from "./button";

export default function ImageResults({
  srcs,
  styleName,
  onClick,
  buttonOnClick,
}) {
  return (
    <div className={`image-results-container ${styleName}`}>
      <SectionTitle title="Results" />
      <div className="result-images-container">
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
        <Button name="Query again" clickHandler={buttonOnClick}></Button>
      </div>
    </div>
  );
}
