<template>
    <div id="app">
        <router-view name="header"></router-view>
        <main>
            <fade-transition origin="center" mode="out-in" :duration="250">
                <router-view/>
            </fade-transition>
        </main>
        <router-view name="footer"></router-view>
    </div>
</template>
<script>
import { FadeTransition } from "vue2-transitions";
import VueAos from 'vue-aos'
// import VueSpeech from 'vue-speech'
// Vue.use(VueSpeech)

export default {
  components: {
    FadeTransition,
    VueAos,
    // VueSpeech
  }
};
</script>
