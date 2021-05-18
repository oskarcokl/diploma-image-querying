export default function PageTitle({ title, styleName }) {
  return <h1 className={`page-title ${styleName}`}>{title}</h1>;
}
