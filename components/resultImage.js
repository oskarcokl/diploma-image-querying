import { useState } from "react";

export default function ResultImage({ src, onClick }) {
  const [selected, setSelected] = useState(false);

  if (src) {
    return <img className="result-image" src={src} onClick={onClick} />;
  } else {
    return <div></div>;
  }
}
