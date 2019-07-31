prefix = '''
<html>
<head>
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/css/bootstrap.min.css">
<script src="https://code.jquery.com/jquery-3.2.1.min.js" integrity="sha256-hwg4gsxgFZhOsEEamdOYGBf13FyQuiTwlAQgxVSNgt4=" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/tether/1.4.0/js/tether.min.js"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/js/bootstrap.min.js" integrity="sha384-vBWWzlZJ8ea9aCX4pEW3rVHjgjt7zpkNpZk+02D9phzyeVkE+jo0ieGizqPLForn" crossorigin="anonymous"></script>
<style>
.unitviz, .unitviz .modal-header, .unitviz .modal-body, .unitviz .modal-footer {
  font-family: Arial;
  font-size: 15px;
}
.unitgrid {
  text-align: center;
  border-spacing: 5px;
  border-collapse: separate;
}
.unitgrid .info {
  text-align: left;
}
.unitgrid .layername {
  display: none;
}
.unitlabel {
  font-weight: bold;
  font-size: 150%;
  text-align: center;
  line-height: 1;
}
.lowscore .unitlabel {
   color: silver;
}
.thumbcrop {
  overflow: hidden;
  width: 540px;
  height: 108px;
}
.unit {
  display: inline-block;
  background: white;
  padding: 3px;
  margin: 2px;
  box-shadow: 0 5px 12px grey;
}
.iou {
  display: inline-block;
  float: right;
}
.modal .big-modal {
  width:auto;
  max-width:90%;
  max-height:80%;
}
.modal-title {
  display: inline-block;
}
.footer-caption {
  float: left;
  width: 100%;
}
.histogram {
  text-align: center;
  margin-top: 3px;
}
.img-wrapper {
  text-align: center;
}
.big-modal img {
  max-height: 60vh;
}
.img-scroller {
  overflow-x: scroll;
}
.img-scroller .img-fluid {
  max-width: initial;
}
.gridheader {
  font-size: 12px;
  margin-bottom: 10px;
  margin-left: 30px;
  margin-right: 30px;
}
.gridheader:after {
  content: '';
  display: table;
  clear: both;
}
.sortheader {
  float: right;
  cursor: default;
}
.layerinfo {
  float: left;
}
.sortby {
  text-decoration: underline;
  cursor: pointer;
}
.sortby.currentsort {
  text-decoration: none;
  font-weight: bold;
  cursor: default;
}

.explain_part{
	width: 105px;
	display: inline-block;
}
</style>
</head>
<body class="unitviz">
<div class="container-fluid">
<h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Prediction Explanation on Places365 Given by Dynamic Model</h3>
<div class="gridheader">
'''

def img_html(img_id, label, info):
    img_content = '''
    <div class="unit">
        <div class="unitlabel">Image val-{}</div>
        <div class="info">
            <span class="explain_part">100% &nbsp;&nbsp;= </span> 
            <span class="explain_part">{:.2f}% &nbsp;&nbsp;+</span> 
            <span class="explain_part">{:.2f}% &nbsp;&nbsp;+</span> 
            <span class="explain_part">{:.2f}% &nbsp;&nbsp;+</span>
            <span class="explain_part">{:.2f}% &nbsp;&nbsp;</span>
        </div>
        <div class="info">
            <span class="explain_part">Prediction: </span> 
            <span class="explain_part">unit {}</span> 
            <span class="explain_part">unit {}</span> 
            <span class="explain_part">unit {}</span>
            <span class="explain_part">unit {}</span>
        </div>
        <div class="info">
            <span class="explain_part">{}</span> 
            <span class="explain_part">{}</span> 
            <span class="explain_part">{}</span> 
            <span class="explain_part">{}</span>
            <span class="explain_part">{}</span>
        </div>
        <div class="thumbcrop">
            <img src="vis/cam_{}.jpg" height="108">
        </div>
    </div>
    '''.format(img_id,  info[0][-1], info[1][-1], info[2][-1], info[3][-1],
                        info[0][0], info[1][0], info[2][0], info[3][0],
               label,   info[0][1], info[1][1], info[2][1], info[3][1], img_id)
    return img_content


suffix = '''
</div>
<div class="modal" id="lightbox">
  <div class="modal-dialog big-modal" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title"></h5>
        <button type="button" class="close"
             data-dismiss="modal" aria-label="Close">
          <span aria-hidden="true">&times;</span>
        </button>
      </div>
      <div class="modal-body">
        <div class="img-wrapper img-scroller">
          <img class="fullsize img-fluid">
        </div>
      </div>
      <div class="modal-footer">
        <div class="footer-caption">
        </div>
      </div>
    </div>
  </div>
</div>

</body>
</html>
'''