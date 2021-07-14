<template>
    <section class="section section-components pb-0" id="section-components">
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-lg-12">
                    <headline style="font: 7.4em 'Montserrat';" :content= "'Describe a Face!'"></headline>
                    <main-menu></main-menu>
                </div>
            </div>
        </div>
    </section>

</template>
<script>
import Headline from './Headline.vue';
import MainMenu from './MainMenu.vue';
export default {
    components: {
        MainMenu,
        Headline
    },
  
};
</script>
<style>

</style>
