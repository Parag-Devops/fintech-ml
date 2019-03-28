$(function() {
    $('#btnCheckFraud').click(function() {
        $.ajax({
            url: '/checkFraud',
            data: $('form').serialize(),
            type: 'POST',
            success: function(response) {
                console.log(response);
            },
            error: function(error) {
                console.log(error);
            }
        });
    });
});