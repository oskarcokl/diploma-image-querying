import Image from "../components/image";
import Container from "../components/container";
import Button from "../components/button";
const axios = require("axios");

API = "http://localhost:8888/";

export default function UploadImage(params) {
  function formUploadHandler(e) {
    e.preventDefault();
    console.log("Yo you just uploaded a form my guy.");

    axios;
  }

  return (
    <Container>
      <form>
        <input type="file" />
        <Button name="Upload image" clickHandler={formUploadHandler}></Button>
      </form>
    </Container>
  );
}
