// JavaScript to change the content based on button click
function changeContent(place) {
  var imgElement = document.querySelector('#image-text-container div img');
  var textElement = document.querySelector('#image-text-container > blockquote');
  var buttonElement = document.querySelectorAll('#image-text-container div button');
  var placeReview = document.querySelectorAll('.place-review');

  // Here you can add the image URLs and text for each place
  var images = {
    'Place1': 'static/img/place_recommend/place1.jpg',
    'Place2': 'static/img/place_recommend/place2.jpg',
    'Place3': 'static/img/place_recommend/place3.jpg',
    'Place4': 'static/img/place_recommend/place4.jpg',
    'Place5': 'static/img/place_recommend/place5.jpg',
  };

  var texts = {
    'Place1': "Chow House is a highly recommended Sichuan restaurant, which aligns with Peng's background as he grew up in Sichuan. The restaurant offers authentic Sichuan food, which Peng might be familiar with and enjoy. The restaurant also has good seating, decoration, and friendly service, which would make for a pleasant dining experience. However, some dishes received mixed reviews, which is why the rating is not a perfect 10.",
    'Place2': 'This place offers Chinese food, which Peng might be familiar with and enjoy as he is from Chengdu, Sichuan. It is also located in the NYU neighborhood, which is convenient for him. The affordable prices are suitable for a student budget. However, the limited seating might be a slight inconvenience if Peng prefers to dine in.',
    'Place3': 'This place is highly recommended by customers for its food, atmosphere, and service. It is also located near NYU, which is convenient for Peng. However, it does not specify if it serves Sichuan cuisine, which Peng might prefer as he grew up in Sichuan.',
    'Place4': 'While the restaurant offers high-quality sushi and a great dining experience, it may not be suitable for Peng who is a first-year undergraduate student and might not be able to afford such an expensive meal. ',
    'Place5': "Dos Toros is a fast food restaurant which could be a good choice for a student like Peng who might be looking for a quick meal. The restaurant offers a variety of options including vegetarian, which could cater to different dietary preferences. However, there are some negative reviews about customer service and portion size, which might affect Peng's dining experience.",
  };

  // Create a new img element
  imgElement.src = images[place];

  // Update the text content
  textElement.textContent = texts[place];
  
  // Update the button status
  let index = 0;
  for (var cur_place in images) {
    buttonElement[index].className = buttonElement[index].className.replace("active-button", "");
    placeReview[index].style.display = "none";

    if (cur_place == place) {
      buttonElement[index].className += "active-button";
      placeReview[index].style.display = "block";
    }
    index++;
  }
}

// image slider

function plusSlides(container_id, n, aside_class = null ) {
  showSlides(container_id, slideIndex += n, aside_class);
}

function currentSlide(container_id, n, aside_class = null ) {
  showSlides(container_id, slideIndex = n, aside_class);
}

function showSlides(container_id, n, aside_class = null ) {
  let i;
  let slides = document.querySelectorAll("#" + container_id + " .my-slides");
  let dots = document.querySelectorAll("#" + container_id + " .demo");
  let captionText = document.querySelector("#" + container_id + " #caption");
  if (n > slides.length) {slideIndex = 1}
  if (n < 1) {slideIndex = slides.length}

  if (aside_class != null) {
    let asides = document.querySelectorAll("." + aside_class);
    for (i = 0; i < asides.length; i++) {
      asides[i].style.display = "none";
    }
    asides[slideIndex-1].style.display = "block";
  }

  for (i = 0; i < slides.length; i++) {
    slides[i].style.display = "none";
  }
  for (i = 0; i < dots.length; i++) {
    dots[i].className = dots[i].className.replace(" active-slide-img", "");
  }
  slides[slideIndex-1].style.display = "block";
  if (dots.length > 0) {
    dots[slideIndex-1].className += " active-slide-img";
    captionText.innerHTML = dots[slideIndex-1].alt;
  }
}


function showInfo(infoId) {
  // First hide all asides
  document.querySelectorAll('.system-fundamental-aside').forEach(function(info) {
    info.style.display = 'none';
  });

  // Then show the clicked one
  var info = document.getElementById(infoId);
  info.style.display = 'block';
}


function setLayoutOfText(className) {
  var elements = document.querySelectorAll('.text');
  for (i = 0; i < elements.length; i++) {
    elements[i].className = elements[i].className.replace("text", className);
  }
}

function showTakeaway(id) {
    // First hide all takeaways
    document.querySelectorAll('.exemplar-takeaways > .takeaway-card').forEach(function(takeaway) {
        takeaway.style.display = 'none';
    });

    // Then show the hovered one
    var takeaway = document.getElementById(id);
    takeaway.style.display = 'block';
}