export default function Navbar({ styleName }) {
  return (
    <div className={`navbar ${styleName}`}>
      <a href="#" className={`navbarElement`}>
        query by example
      </a>
      <a href="#" className={`navbarElement`}>
        add images to index
      </a>
    </div>
  );
}
