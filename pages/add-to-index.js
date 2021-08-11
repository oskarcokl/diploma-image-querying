import Head from "next/dist/next-server/lib/head";
import Container from "../components/container";
import Navbar from "../components/navbar";
import PageTitle from "../components/pageTitle";
import FileUpload from "../components/fileUpload";
import AddImages from "../components/addImages";
import Button from "../components/button";
import { useState } from "react";

export default function AddToIndex(params) {
  const [addUrls, setAddUrls] = useState([]);

  const onChangeHandler = async (e) => {
    const urls = [];
    for (let file of e.target.files) {
      urls.push(URL.createObjectURL(file));
    }
    setAddUrls(urls);
  };

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
            <AddImages srcs={addUrls} styleName="query-example-grid-results" />
            <FileUpload
              additionalClasses=""
              name={"file"}
              text={"select images"}
              onChangeHandler={onChangeHandler}
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
