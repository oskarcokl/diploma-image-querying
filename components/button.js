export default function Button({ name, clickHandler }) {
  return (
    <button onClick={clickHandler} className="button">
      {name}
    </button>
  );
}
