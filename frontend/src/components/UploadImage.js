// UploadImage.js

import React, { useState } from 'react';

function UploadImage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [message, setMessage] = useState('');
  
  const handleImageUpload = async (event) => {
    const image = event.target.files[0];
    
    const formData = new FormData();
    formData.append('image', image);
    
    try {
      const response = await fetch('http://localhost:8000/recognizer/', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      setMessage(data.message);
      
      // Decode the base64 encoded image
      const decodedImage = atob(data.image);
      
      // Create a Blob from the decoded image data
      const blob = new Blob([decodedImage], { type: 'image/jpeg' });
      
      // Create a temporary URL for the Blob
      const imageUrl = URL.createObjectURL(blob);
      
      setSelectedImage(imageUrl);
    } catch (error) {
      console.error('Error:', error);
    }
  };
  
  return (
    <div>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      {selectedImage && <img src={selectedImage} alt="Processed Image" />}
      {message && <p>{message}</p>}
    </div>
  );
}

export default UploadImage;
