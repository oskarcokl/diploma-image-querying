export default function UploadedImage({ src }) {
  return (
    <div>
      <img className="uploaded-image" src={src} />
    </div>
  );
}
