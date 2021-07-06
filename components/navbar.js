export default function Navbar({ styleName }) {
  return (
    <div className={`navbar ${styleName}`}>
      <a href="query-by-example" className={`navbarElement`}>
        query by example
      </a>
      <a href="add-to-index" className={`navbarElement`}>
        add images to index
      </a>
    </div>
  );
}
