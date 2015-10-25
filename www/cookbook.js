'use strict';

$(document).ready(function() {
    /* GLOBAL STATE */
    /* The index.json content as returned from the server */
    var data_json = {};

    function network_error(ajax, status, error) {
        $("#error-message").text(
            "Error fetching content. " +
            "Perhaps web server has gone down.");
        $("#error").modal('show');
    }

    function draw_page() {
        var tags = data_json['tags'];
        var titles = data_json['titles'];
        var tag_counts = {};

        $.each(tags, function(key, tag_list) {
            $.each(tag_list, function(i, tag) {
                if (!tag_counts[tag]) {
                    tag_counts[tag] = 1;
                }
                else {
                    tag_counts[tag] += 1;
                }
            });
        });

        tag_counts['Untagged'] = 0;

        var tag_sets = {};

        $.each(titles, function(fn, title) {
            var tagset = tags[fn];
            if (!tagset) {
                tagset = ['Untagged'];
            }

            var counted_tags = [];
            $.each(tagset, function(i, tag) {
                counted_tags.push([tag_counts[tag], tag]);
            });
            counted_tags.sort(function(a, b) {
                return b[0] - a[0];
            });

            var tag_id = '';
            $.each(counted_tags, function(i, tag) {
                if (i == 0) {
                    tag_id = tag[1];
                }
                else {
                    tag_id = tag_id + " / " + tag[1];
                }
            });

            if (!tag_sets[tag_id]) {
                tag_sets[tag_id] = [[fn, title, tag_counts[counted_tags[0][1]]]];
            }
            else {
                tag_sets[tag_id].push([fn, title, tag_counts[counted_tags[0][1]]]);
            }
        });

        var tag_set_list = [];
        $.each(tag_sets, function(key, value) {
            tag_set_list.push([key, value]);
        });
        tag_set_list.sort();

        var body = $("#display-body");
        var div = $("<div class='container'>");

        $.each(tag_set_list, function(i, key_value) {
            var tag_id = key_value[0];
            var items = key_value[1];
            items.sort();

            var box = $("<div class='row'>");
            var title = $("<h3>");
            title.text(tag_id);
            box.append(title);

            var ul = $("<ul class='list-group'>");
            $.each(items, function(i, item) {
                var li = $("<li class='list-group-item pull-left'>");
                var a = $("<a target='_blank'>");
                a.attr('href',
                       "http://nbviewer.ipython.org/github/pv/SciPy-CookBook/tree/master/ipython/" + item[0]);
                a.text(item[1]);
                li.append(a);
                ul.append(li);
            });
            box.append(ul);

            div.append(box);
        });
        
        body.append(div);
    }

    function init() {
        $.ajax({
            url: "data.json", dataType: "json"
        }).done(function (data) {
            data_json = data;
            draw_page();
        }).fail(function () {
            $.ajax({
                url: "https://github.com/pv/SciPy-CookBook/raw/master/www/data.json", dataType: "json"
            }).done(function (data) {
                data_json = data;
                draw_page();
            }).fail(function () {
                network_error();
            });
        });
    }


    /* Launch */
    init();
});
