import Image from "../components/image";
import Container from "../components/container";
import Button from "../components/button";
import { useState } from "react";
import axios from "axios";

const API = "http://localhost:8888/";

export default function UploadImage(params) {
  const [selectedFile, setSelectedFile] = useState(null);

  const onClickHandler = (e) => {
    e.preventDefault();

    const data = new FormData();
    data.append("file", selectedFile);

    console.log(e);

    console.log("Yo you just uploaded a form my guy.");

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
    setSelectedFile(event.target.files[0]);
    console.log(event.target.files[0]);
  };

  return (
    <Container>
      <form>
        <input type="file" name="file" onChange={onChangeHandler} />
        <Button name="Upload image" clickHandler={onClickHandler}></Button>
      </form>
    </Container>
  );
}
