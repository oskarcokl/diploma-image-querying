import Container from "../components/container";
import Button from "../components/button";
import PageTitle from "../components/pageTitle";
import { useState } from "react";
import axios from "axios";
import Head from "next/head";
import FileUpload from "../components/fileUpload";
import UploadedImage from "../components/uploadedImage";
import ImageResults from "../components/imageResults";
import Navbar from "../components/navbar";
import SectionTitle from "../components/sectionTitle";

const API = "http://localhost:8888/";

export default function UploadImage(params) {
  const [selectedFiles, setSelectedFiles] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [resultImages, _setResultImages] = useState([]);

  const setResultImages = (src) => {
    const temp = [];
    for (let i = 0; i < 10; i++) temp.push(src);

    _setResultImages(temp);
  };

  const onClickHandler = (e) => {
    e.preventDefault();

    console.log("Uploading images to server.");

    const data = new FormData();
    for (let i = 0; i < selectedFiles.length; i++) {
      data.append("file", selectedFiles[i]);
    }

    axios
      .post(API + "file-upload", data, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then((res) => {
        console.log(res.statusText);
      });
  };

  const onChangeHandler = async (event) => {
    const exampleImageURL = URL.createObjectURL(event.target.files[0]);

    setResultImages(exampleImageURL);

    setImageSrc(exampleImageURL);
    setSelectedFiles(event.target.files);
  };

  return (
    <div>
      <Head>
        <title>QueryByExample</title>
      </Head>
      <Container>
        <div className="query-exmaple-grid-container">
          <Navbar styleName="query-example-grid-navbar"></Navbar>
          <PageTitle
            title="Query By Example"
            styleName="query-example-grid-header"
          />
          <form className="image-upload-container query-example-grid-query">
            <SectionTitle title="Upload example image" />
            <UploadedImage src={imageSrc} />
            <FileUpload
              additionalClasses=""
              name={"file"}
              onChangeHandler={onChangeHandler}
            ></FileUpload>
            <Button
              additionalClasses="flex-none"
              name="Upload image"
              clickHandler={onClickHandler}
            ></Button>
          </form>
          <ImageResults
            styleName="query-example-grid-results"
            srcs={resultImages}
          ></ImageResults>
        </div>
      </Container>
    </div>
  );
}
