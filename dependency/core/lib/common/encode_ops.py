class EncodeOps:
    @staticmethod
    def encode_image(image):
        import numpy as np
        import cv2
        import base64
        """
        Encodes an image to a base64 string with a data URL prefix.

        Parameters:
            image (numpy.ndarray): The image to encode, typically read by OpenCV (in BGR format).

        Returns:
            str: A base64-encoded string representing the image, prefixed with "data:image/jpg;base64,".

        Raises:
            ValueError: If the image encoding fails or the input is not a valid numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array.")

        # Encode the image to JPEG format
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Image encoding failed.")

        # Convert the encoded image to base64
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpg;base64,{base64_str}"

    @staticmethod
    def decode_image(encoded_img_str):
        import numpy as np
        import cv2
        import base64
        """
        Decodes a base64-encoded image string back into an OpenCV image (numpy array).

        Parameters:
            encoded_img_str (str): The base64 string representing the image.
                                   It should include the "data:image/jpg;base64," prefix.

        Returns:
            numpy.ndarray: The decoded image in BGR format suitable for OpenCV processing.

        Raises:
            ValueError: If decoding fails due to an invalid base64 string or incompatible data.
        """
        if not isinstance(encoded_img_str, str):
            raise ValueError("Encoded image must be a string.")

        # Remove the data URL prefix if present
        if encoded_img_str.startswith("data:image"):
            encoded_img_str = encoded_img_str.split(',', 1)[1]

        try:
            # Decode the base64 string to bytes
            encoded_img_bytes = base64.b64decode(encoded_img_str)
            # Convert bytes to a numpy array and decode to an OpenCV image
            decoded_img = cv2.imdecode(np.frombuffer(encoded_img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if decoded_img is None:
                raise ValueError("Image decoding failed; data may be corrupt or incompatible.")
            return decoded_img
        except (base64.binascii.Error, ValueError) as e:
            raise ValueError("Invalid base64 string or incompatible image data.") from e
