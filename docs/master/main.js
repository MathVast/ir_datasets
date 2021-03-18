
function scrollIntoViewIfNeeded(target) { 
    if (target.getBoundingClientRect().bottom > window.innerHeight) {
        target.scrollIntoView(false);
    }

    if (target.getBoundingClientRect().top < 0) {
        target.scrollIntoView();
    } 
}
function selectText(element) {
    if (document.selection) { // IE
        var range = document.body.createTextRange();
        range.moveToElementText(element);
        range.select();
    } else if (window.getSelection) {
        var range = document.createRange();
        range.selectNode(element);
        window.getSelection().removeAllRanges();
        window.getSelection().addRange(range);
    }
}
$(document).ready(function() {
    $('.ds-ref').click(function () {
        var target = $('[id="' + $(this).text() + '"]');
        target.effect("highlight", {}, 1000);
        scrollIntoViewIfNeeded(target[0]);
    });
    var left = 0, top = 0;
    $(document).on('mousedown', '.select', function(e) {left = e.pageX; top = e.pageY;});
    $(document).on('mouseup', '.select', function(e) { if (left == e.pageX && top == e.pageY) {selectText(this);} });
    $(document).on('click', '.jumpto', function(e) {
        var target = $('[id="' + $(e.target).attr('href').substr(1) + '"]');
        target.effect("highlight", {}, 1000);
        scrollIntoViewIfNeeded(target[0]);
        return false;
    });
    $('.tabs').each(function (i, e) {
        var first = $(e).find('.tab-content:first');
        $(e).find('.tab-content').hide();
        first.show();
        $(e).find('.tab:first').addClass('selected');
        $(e).find('.tab').prependTo(e);
    });
    $(document).on('click', '.tab', function(e) {
        var $target = $(e.target);
        var tabs = $target.closest('.tabs');
        tabs.find('.tab-content').hide();
        $('[id="'+$target.attr('target')+'"]').show();
        tabs.find('.tab.selected').removeClass('selected');
        $target.addClass('selected');
    });
});
function toEmoji(test) {
    if (test) {
        return '✅';
    }
    return '❌';
}
function toTime(duration) {
    if (duration < 60) {
        return duration.toFixed(2) + 's';
    }
    var minutes = Math.floor(duration / 60);
    var seconds = duration % 60;
    return minutes.toFixed(0) + 'm ' + seconds.toFixed(0) + 's';
}
function generateDownloads(title, downloads) {
    if (downloads.length === 0) {
        return $('<div></div>');
    }
    var allGood = true;
    var $content = $('<table></table>');
    $content.append($('<tr><th>Avail</th><th>ID</th><th>Time</th><th>URL</th><th>Last Tested At</th><th>Expected MD5 Hash</th></tr>'));
    var goodCount = 0;
    var totalCount = 0;
    $.each(downloads, function (i, dl) {
        var good = dl.result === "PASS";
        totalCount += 1;
        if (!good) {
            allGood = false;
        } else {
            goodCount += 1;
        }
        $content.append($('<tr></tr>')
            .append($('<td></td>').text(toEmoji(good)).attr('title', dl.result))
            .append($('<td></td>').text(dl.name))
            .append($('<td></td>').text(toTime(dl.duration)))
            .append($('<td></td>').append($('<a>').attr('href', dl.url).text('link')))
            .append($('<td></td>').text(dl.time.substring(0, 19)))
            .append($('<td></td>').text(dl.md5))
        );
    });
    return $('<details></details>')
        .append($('<summary></summary>').text(toEmoji(allGood) + ' ' + title + ' (' + goodCount.toString() + ' of ' + totalCount.toString() + ')'))
        .append($('<p>These files are automatically downloaded by ir_datasets as they are needed. We also periodically check that they are still available and unchanged through an automated <a href="https://github.com/allenai/ir_datasets/actions/workflows/verify_downloads.yml">GitHub action</a>. The latest results from that test are shown here:</p>'))
        .append($content)
        .prop('open', !allGood);
}
