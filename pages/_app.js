import "../styles/globals.css";
import "../styles/Navbar.css";
import "../styles/Button.css";
import "../styles/FileUploadElement.css";
import "../styles/QueryByExamplePage.css";
import "../styles/UploadedImage.css";
import "../styles/ResultImage.css";
import "../styles/ImageResults.css";
import "../styles/SectionTitle.css";
import "react-toastify/dist/ReactToastify.css";

import { ToastContainer } from "react-toastify";

function MyApp({ Component, pageProps }) {
  return (
    <div>
      <Component {...pageProps} />
      <ToastContainer />
    </div>
  );
}

export default MyApp;
