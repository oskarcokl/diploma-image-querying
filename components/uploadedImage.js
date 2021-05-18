import Image from "next/image";

export default function UploadedImage({ src }) {
  return (
    <div>
      <img className="uploaded-image" src="/placeholder.jpg" />
    </div>
  );
}
