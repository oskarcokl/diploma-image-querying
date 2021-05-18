export default function PageTitle({ title, styleName }) {
  return <h1 className={`text-3xl text-center ${styleName}`}>{title}</h1>;
}
