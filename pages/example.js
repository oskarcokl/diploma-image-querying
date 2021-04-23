import Button from "../components/button";
import Container from "../components/container";
import Image from "../components/image";
const axios = require("axios");

export default function Example(params) {
  let imageUrl = "https://unsplash.com/photos/eqQZGX4-X_A";
  async function handleClick(e) {
    e.preventDefault();
    console.log("The button was clicked, why did you do that :(");

    try {
      const response = await axios.get("http://localhost:3004/images");
      console.log(response.data[0].url);

      imageUrl = response.data[0].url;
    } catch (error) {
      console.log(error);
    }
  }

  return (
    <Container>
      <Image imgSrc={imageUrl}></Image>
      <Button name="Please dont click me" clickHandler={handleClick}></Button>
    </Container>
  );
}
