import React, { useState } from 'react';
import axios from 'axios';

function ImageUploader() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [data,setData] = useState(null);
  const [processedImage,setProcessedImage] = useState(null);
  const [result,setResult] = useState(null);
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedImage(URL.createObjectURL(file));
    setSelectedFile(file);
    setProcessedImage(null);
    setResult(null);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    const formData = new FormData();
    formData.append('image', selectedFile);
    try {
        const endpoint = 'http://localhost:8000/recognizer/';
        const inresponse = await axios.post(endpoint, formData);
        setData(inresponse.data)
        setResponse(inresponse.data.message);

      // Decode the base64 encoded image
      setProcessedImage(inresponse.data.image);
      setResult(inresponse.data.result)
      //const decodedImage = atob(data.image);
      
      // Create a Blob from the decoded image data
      //const blob = new Blob([decodedImage], { type: 'image/jpeg' });
      
      // Create a temporary URL for the Blob
      //const imageUrl = URL.createObjectURL(blob);
      //console.log(imageUrl)
      //setSelectedImage(imageUrl);

    } catch (error) {
      console.error(error);
    }
    setLoading(false);
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} />
        <button type="submit">Upload</button>
      </form>
      {loading && <div>Loading...</div>}
      {response && <div>{response}</div>}
      {selectedImage && <img src={selectedImage} alt="Uploaded" />}
      {processedImage && <img src={`data:image/jpg;base64,${processedImage}`} alt="Processed" />}
      {result && <div>{result}</div>}
    </div>
  );
}

export default ImageUploader;
