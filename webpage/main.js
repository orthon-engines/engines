$(document).ready(function() {
  
  // Accordion menu toggle
  $('#menu span.opener').on('click', function() {
    $(this).toggleClass('active');
    $(this).next('ul').slideToggle(200);
  });
  
  // Keep current section open based on URL
  var currentPath = window.location.pathname;
  $('#menu ul ul a').each(function() {
    var linkPath = $(this).attr('href');
    if (currentPath.indexOf(linkPath) !== -1 && linkPath !== '/') {
      $(this).closest('ul').show();
      $(this).closest('ul').prev('.opener').addClass('active');
      $(this).css('color', '#f56a6a');
    }
  });
  
});
