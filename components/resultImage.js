export default function ResultImage({ src }) {
  if (src) {
    return <img className="result-image" src={src} />;
  } else {
    return <div></div>;
  }
}
