import Image from "../components/image";
import Container from "../components/container";
import Button from "../components/button";
import { useState } from "react";
import axios from "axios";

const API = "http://localhost:8888/";

export default function UploadImage(params) {
  const [selectedFiles, setSelectedFiles] = useState(null);

  const onClickHandler = (e) => {
    e.preventDefault();

    console.log("Uploading images to server.");

    const data = new FormData();
    for (let i = 0; i < selectedFiles.length; i++) {
      //console.log(selectedFiles[i]);
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

  const onChangeHandler = (event) => {
    setSelectedFiles(event.target.files);
  };

  return (
    <Container>
      <form>
        <input type="file" name="file" multiple onChange={onChangeHandler} />
        <Button name="Upload image" clickHandler={onClickHandler}></Button>
      </form>
    </Container>
  );
}
