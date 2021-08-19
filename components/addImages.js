import ResultImage from "./resultImage";
import SectionTitle from "./sectionTitle";

export default function ImageResults({ srcs, styleName }) {
  if (srcs) {
    const addImages = [];
    for (let i = 0; i < srcs.length; i++) {
      addImages.push(<ResultImage src={srcs[i]} key={i} />);
    }

    return (
      <div className={`image-results-container ${styleName}`}>
        <div className="result-images-container">{addImages}</div>
      </div>
    );
  } else {
    return <div></div>;
  }
}
