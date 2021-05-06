export default function Button({ name, clickHandler, additionalClasses }) {
  let className =
    "bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded-full";

  if (additionalClasses) {
    console.log(additionalClasses);
    className += " " + additionalClasses;
  }

  return (
    <button onClick={clickHandler} className={className}>
      {name}
    </button>
  );
}
