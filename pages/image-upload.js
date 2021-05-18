import Container from "../components/container";
import Button from "../components/button";
import PageTitle from "../components/pageTitle";
import { useState } from "react";
import axios from "axios";
import Head from "next/head";
import FileUpload from "../components/fileUpload";
import UploadedImage from "../components/uploadedImage";
import ImageResults from "../components/imageResults";

const API = "http://localhost:8888/";

export default function UploadImage(params) {
  const [selectedFiles, setSelectedFiles] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);

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
    setImageSrc(URL.createObjectURL(event.target.files[0]));
    setSelectedFiles(event.target.files);
  };

  return (
    <div>
      <Head>
        <title>QueryByExample</title>
      </Head>
      <Container>
        <PageTitle title="Query By Example" />
        <form className="image-upload-container">
          <UploadedImage src={imageSrc} />
          <FileUpload
            additionalClasses="mx-10"
            name={"file"}
            onChangeHandler={onChangeHandler}
          ></FileUpload>
          <Button
            additionalClasses="flex-none"
            name="Upload image"
            clickHandler={onClickHandler}
          ></Button>
        </form>
        <ImageResults></ImageResults>
      </Container>
    </div>
  );
}
