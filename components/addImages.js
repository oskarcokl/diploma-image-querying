import ResultImage from "./resultImage";
import SectionTitle from "./sectionTitle";

export default function ImageResults({ srcs, styleName }) {
  if (srcs) {
    const addImages = [];
    for (let i = 0; i < srcs.length; i++) {
      addImages.push(<ResultImage src={srcs[i]} />);
    }

    return (
      <div className={`image-results-container ${styleName}`}>
        <div className="result-images-container">
          <div className="column-1">{addImages}</div>
        </div>
      </div>
    );
  } else {
    return <div></div>;
  }
}
