import Head from "next/dist/next-server/lib/head";
import Container from "../components/container";
import Navbar from "../components/navbar";
import PageTitle from "../components/pageTitle";
import FileUpload from "../components/fileUpload";
import AddImages from "../components/addImages";
import Button from "../components/button";
import { useState } from "react";
import axios from "axios";
import { toast } from "react-toastify";

const API = "http://localhost:8888/";

export default function AddToIndex(params) {
  const [addUrls, setAddUrls] = useState([]);
  const [addImgs, setAddImgs] = useState([]);

  const onChangeHandler = async (e) => {
    const urls = [];
    for (let file of e.target.files) {
      urls.push(URL.createObjectURL(file));
    }
    setAddImgs(e.target.files);
    setAddUrls(urls);
  };

  const onClickHandler = (e) => {
    e.preventDefault();

    // Clear add images
    setAddUrls([]);
    addToIndex(addImgs);
  };

  const toastOnClick = (e) => {
    e.preventDefault();

    toast("Big chungus!");
  };

  const addToIndex = (addImages) => {
    console.log("Uploading images to server.");
    const data = new FormData();
    for (let i = 0; i < addImages.length; i++) {
      data.append("file", addImages[i]);
    }

    axios
      .post(API + "add-index", data, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .catch((e) => {
        console.log(e);
      })
      .then((res) => {
        console.log(res);
      });
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
              clickHandler={toastOnClick}
            ></Button>
          </form>
        </div>
      </Container>
    </div>
  );
}
