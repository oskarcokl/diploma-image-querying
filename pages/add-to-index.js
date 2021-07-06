import Head from "next/dist/next-server/lib/head";
import Container from "../components/container";
import Navbar from "../components/navbar";
import PageTitle from "../components/pageTitle";
import SectionTitle from "../components/sectionTitle";
import FileUpload from "../components/fileUpload";
import UploadedImage from "../components/uploadedImage";
import Button from "../components/button";

export default function AddToIndex(params) {
  return (
    <div>
      <Head>
        <title>QueryByExample</title>
      </Head>
      <Container>
        <div className="query-exmaple-grid-container">
          <Navbar styleName="query-example-grid-navbar"></Navbar>
          <PageTitle
            title="Add images to index"
            styleName="query-example-grid-header"
          />
          <form className="image-upload-container query-example-grid-query">
            <SectionTitle title="Upload example image" />
            <UploadedImage /*src={imageSrc}*/ />
            <FileUpload
              additionalClasses=""
              name={"file"}
              //onChangeHandler={onChangeHandler}
            ></FileUpload>
            <Button
              additionalClasses="flex-none"
              name="Add to index"
              //clickHandler={onClickHandler}
            ></Button>
          </form>
        </div>
      </Container>
    </div>
  );
}
