import Head from "next/dist/next-server/lib/head";
import Container from "../components/container";
import Navbar from "../components/navbar";
import PageTitle from "../components/pageTitle";
import SectionTitle from "../components/sectionTitle";
import FileUpload from "../components/fileUpload";
import UploadedImage from "../components/uploadedImage";
import Button from "../components/button";

export default function AddToIndex(params) {
  const onClickHandler = (e) => {
    e.preventDefault();

    console.log("You clicked a button buddy.");
  };

  return (
    <div>
      <Head>
        <title>QueryByExample</title>
      </Head>
      <Container>
        <div className="add-index-grid-container">
          <Navbar styleName="query-example-grid-navbar"></Navbar>
          <PageTitle
            title="Add images to index"
            styleName="query-example-grid-header"
          />
          <form className="image-upload-container query-example-grid-query">
            <UploadedImage /*src={imageSrc}*/ />
            <FileUpload
              additionalClasses=""
              name={"file"}
              text={"select images"}
            ></FileUpload>
            <Button
              additionalClasses="flex-none"
              name="Add to index"
              clickHandler={onClickHandler}
            ></Button>
          </form>
        </div>
      </Container>
    </div>
  );
}
