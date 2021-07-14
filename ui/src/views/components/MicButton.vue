
<template>
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-lg-12">
                    
                    <!-- mic icon -->
                    <div class='tab6' :class="this.clicked === 0 ? 'tab6' : 'tab6_rec'">
                        <div v-on:click="record" style="color:white">
                            <i class="fa fa-microphone" aria-hidden="true"></i>
                        </div>
                    </div>

                </div>
            </div>
        </div>

</template>




<script>
export default {
    name: "mic-button",
    data: function() {
    return{
        clicked: 0
      }
    },
    mounted(){
        this.clicked = 0;
    },
    methods: {
        record() {
            if (this.clicked == 1){
                this.clicked = 0;
            }
            else{
                this.clicked = 1;
            }
            this.$emit('clicked', this.clicked)
        }
    }
};
</script>

