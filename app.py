from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
)
import cv2
import os
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def remove_watermark_frequency(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Split the image into color channels
    b, g, r = cv2.split(image)

    # Apply Fourier transform to each color channel
    b_fft = np.fft.fft2(b)
    g_fft = np.fft.fft2(g)
    r_fft = np.fft.fft2(r)

    # Create a high-pass filter to remove high-frequency noise
    rows, cols = b.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - 30 : crow + 30, ccol - 30 : ccol + 30] = 0

    # Apply the high-pass filter in frequency domain to each channel
    b_fft_filtered = b_fft * mask
    g_fft_filtered = g_fft * mask
    r_fft_filtered = r_fft * mask

    # Inverse Fourier transform to obtain filtered images
    b_filtered = np.fft.ifft2(b_fft_filtered)
    g_filtered = np.fft.ifft2(g_fft_filtered)
    r_filtered = np.fft.ifft2(r_fft_filtered)

    # Combine the filtered color channels back into an image
    filtered_image = cv2.merge(
        (
            np.abs(b_filtered).astype(np.uint8),
            np.abs(g_filtered).astype(np.uint8),
            np.abs(r_filtered).astype(np.uint8),
        )
    )

    return filtered_image
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Fourier transform to convert the image to frequency domain
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)

    # Create a high-pass filter to remove high-frequency noise
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - 30 : crow + 30, ccol - 30 : ccol + 30] = 0

    # Apply the high-pass filter in frequency domain
    f_shift_filtered = f_shift * mask
    f_transform_filtered = np.fft.ifftshift(f_shift_filtered)
    image_filtered = np.fft.ifft2(f_transform_filtered)
    image_filtered = np.abs(image_filtered)

    return image_filtered.astype(np.uint8)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Process the uploaded image
            processed_image = remove_watermark_frequency(filepath)
            processed_filename = "processed_" + filename
            processed_filepath = os.path.join(
                app.config["UPLOAD_FOLDER"], processed_filename
            )
            cv2.imwrite(processed_filepath, processed_image)

            return redirect(
                url_for(
                    "result",
                    original_image=filename,
                    processed_image=processed_filename,
                )
            )

    return render_template("index.html")


@app.route("/result")
def result():
    original_image = request.args.get("original_image")
    processed_image = request.args.get("processed_image")
    return render_template(
        "result.html", original_image=original_image, processed_image=processed_image
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
