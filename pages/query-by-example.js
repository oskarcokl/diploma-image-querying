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
import { MoonLoader } from "react-spinners";

const API = "http://localhost:8888/";

export default function QueryByExample(params) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [resultImages, _setResultImages] = useState([]);
  const [loading, setLoading] = useState(false);

  const setResultImages = (src) => {
    const temp = [];
    for (let i = 0; i < src.length; i++) temp.push(src[i]);

    _setResultImages(temp);
  };

  const clearResultImages = () => {
    _setResultImages([]);
  };

  const queryByExample = (queryFile) => {
    console.log("Uploading images to server.");

    const data = new FormData();
    data.append("file", queryFile);

    axios
      .post(API + "cbir-query", data, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .catch((e) => {
        console.log(e);
      })
      .then((res) => {
        const imgNames = res.data.result_imgs;
        let returnedImages = [];
        for (let imgName of imgNames) {
          returnedImages.push("/dataset/" + imgName);
        }

        console.log(returnedImages);

        setResultImages(returnedImages);
        setLoading(false);
      });
  };

  const onChangeHandler = async (event) => {
    const exampleImageURL = URL.createObjectURL(event.target.files[0]);

    setImageSrc(exampleImageURL);
    setSelectedFile(event.target.files[0]);
    clearResultImages();
    setLoading(true);
    queryByExample(event.target.files[0]);
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
            {/* <UploadedImage src={imageSrc} /> */}
            <MoonLoader
              color={"#45a191"}
              loading={loading}
              size={125}
              speedMultiplier={0.5}
            />
            <FileUpload
              additionalClasses=""
              name={"file"}
              onChangeHandler={onChangeHandler}
              text={"select image"}
            ></FileUpload>
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
