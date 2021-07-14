
<template>
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-lg-12">
                    <svg viewBox="0 0 960 300">
                        <symbol id="s-text">
                            <text text-anchor="middle" x="50%" y="80%">{{content}}</text>
                        </symbol>
                                              

                        <g class = "g-ants">
                            <use xlink:href="#s-text" class="text-copy"></use>
                            <use xlink:href="#s-text" class="text-copy"></use>
                            <use xlink:href="#s-text" class="text-copy"></use>
                            <use xlink:href="#s-text" class="text-copy"></use>
                            <use xlink:href="#s-text" class="text-copy"></use>
                        </g>
                    </svg>


                </div>
            </div>
        </div>
</template>




<script>
export default {
  name: "headline",
  data: function() {
      return{
          content_text:''
      }
  },
  props:{
      content: {
        type: String,
        required: true,
  	},
  },
  mounted(){
      this.content_text= this.content
      console.log(this.content_text)

  }


};
</script>
<style>
</style>
