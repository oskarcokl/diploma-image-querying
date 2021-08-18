import Container from "../components/container";
import Button from "../components/button";
import PageTitle from "../components/pageTitle";
import { useState } from "react";
import axios from "axios";
import Head from "next/head";
import FileUpload from "../components/fileUpload";
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
  const [selectedImages, setSelectedImages] = useState({});

  const setResultImages = (src) => {
    const temp = [];
    for (let i = 0; i < src.length; i++) temp.push(src[i]);

    _setResultImages(temp);
  };

  const clearResultImages = () => {
    _setResultImages([]);
  };

  const onClickImgHandler = (e) => {
    const target = e.target;
    target.classList.toggle("selected");
    const imageName = target.src.split("/")[4];
    toggleSelectedImage(imageName);
  };

  const onClickQueryAgainHandler = () => {
    setLoading(true);
    queryRocchio();
  };

  const toggleSelectedImage = (imageName) => {
    selectedImages[imageName].selected = !selectedImages[imageName].selected;
  };

  const queryRocchio = () => {
    console.log("Querying again with ROCCHIO.");

    const data = new FormData();
    data.append("selectedImages", JSON.stringify(selectedImages));

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
        parseResults(res);
      });
  };

  const queryByExample = (queryFile) => {
    console.log("Uploading images to server.");

    const data = new FormData();
    data.append("file", queryFile);

    data.append("selectedImages", JSON.stringify(selectedImages));

    axios
      .post(API + "cbir-query", data, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .catch((e) => {
        console.log(e);
      })
      .then((res) => parseResults(res));
  };

  const parseResults = (res) => {
    console.log(res);
    const imgNames = res.data.ordered_result;

    setSelectedImages(res.data.dict);

    let returnedImages = [];
    for (let imgName of imgNames) {
      returnedImages.push("/dataset/" + imgName);
    }

    setResultImages(returnedImages);
    setLoading(false);
  };

  const onChangeHandler = async (event) => {
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
            {!selectedFile && <SectionTitle title="Upload example image" />}

            <MoonLoader
              color={"#45a191"}
              loading={loading}
              size={125}
              speedMultiplier={0.5}
            />
            {!loading && (
              <FileUpload
                additionalClasses=""
                name={"file"}
                onChangeHandler={onChangeHandler}
                text={"select image"}
              ></FileUpload>
            )}
          </form>

          {resultImages.length > 0 && (
            <ImageResults
              styleName="query-example-grid-results"
              srcs={resultImages}
              onClick={(e) => onClickImgHandler(e)}
            ></ImageResults>
          )}
        </div>
      </Container>
    </div>
  );
}
