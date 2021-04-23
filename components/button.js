export default function Button({ name, clickHandler }) {
  return (
    <button
      onClick={clickHandler}
      className="bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded-full"
    >
      {name}
    </button>
  );
}
