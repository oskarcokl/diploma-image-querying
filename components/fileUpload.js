export default function FileUpload({
  name,
  onChangeHandler,
  additionalClasses,
  text,
}) {
  let className;
  if (additionalClasses) {
    className += " " + additionalClasses;
  }

  return (
    <div className={className}>
      <label className="file-upload-label">
        <svg
          className="file-upload-svg"
          fill="currentColor"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 20 20"
        >
          <path d="M16.88 9.1A4 4 0 0 1 16 17H5a5 5 0 0 1-1-9.9V7a3 3 0 0 1 4.52-2.59A4.98 4.98 0 0 1 17 8c0 .38-.04.74-.12 1.1zM11 11h3l-4-4-4 4h3v3h2v-3z" />
        </svg>
        <span className="file-upload-span">{text}</span>
        <input
          type="file"
          name={name}
          className="hidden"
          multiple
          onChange={onChangeHandler}
        />
      </label>
    </div>
  );
}
