$(document).ready(function () {
    // Handle Train Model Form
    $("#trainForm").on("submit", function (e) {
        e.preventDefault();
        const formData = new FormData(this);

        fetch("/train", {
            method: "POST",
            body: formData,
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.error) {
                    $("#trainResponse").html(`<div class="alert alert-danger">${data.error}</div>`);
                } else {
                    $("#trainResponse").html(`<div class="alert alert-success">${data.message}</div>`);
                }
            });
    });

    // Handle Generate Synthetic Data Form
    $("#generateForm").on("submit", function (e) {
        e.preventDefault();
        const formData = new FormData(this);

        fetch("/generate", {
            method: "POST",
            body: formData,
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.error) {
                    $("#generateResponse").html(`<div class="alert alert-danger">${data.error}</div>`);
                } else {
                    $("#generateResponse").html(`<div class="alert alert-success">Synthetic Data Generated Successfully!</div>`);
                    console.log(data.synthetic_data);
                }
            });
    });
});
