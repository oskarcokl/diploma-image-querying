export default function UploadedImage({ src }) {
  if (src) {
    return (
      <div>
        <img className="uploaded-image" src={src} />
      </div>
    );
  } else {
    return <div></div>;
  }
}
